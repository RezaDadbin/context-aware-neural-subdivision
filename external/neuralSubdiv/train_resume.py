from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import pickle
import random
import math
import hashlib
import sys
import time

import numpy as np
import torch

from include import random3DRotation, tgp
from models import SubdNet
from validate_dataset import require_valid_dataset

NETPARAMS = 'netparams.dat'
CHECKPOINT = 'checkpoint_latest.pt'
RUN_MANIFEST = 'run_manifest.json'


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def build_run_manifest(params, model_kind):
    transient = {'device', 'epochs', 'output_path'}
    artifact_keys = {
        'train_pkl', 'valid_pkl', 'initial_checkpoint', 'core_checkpoint'}
    stable_params = {
        key: value for key, value in params.items()
        if key not in transient and key not in artifact_keys
    }
    artifacts = {}
    for key in sorted(artifact_keys):
        path = params.get(key)
        if path:
            if not os.path.isfile(path):
                raise FileNotFoundError('%s does not exist: %s' % (key, path))
            artifacts[key] = file_sha256(path)

    model_files = ['models.py', 'include.py']
    if model_kind == 'context':
        model_files.append('context_models.py')
    code = {
        name: file_sha256(os.path.join(os.path.dirname(__file__), name))
        for name in model_files
    }
    identity = {
        'schema': 1,
        'model_kind': model_kind,
        'parameters': stable_params,
        'artifacts_sha256': artifacts,
        'model_code_sha256': code,
    }
    encoded = json.dumps(
        identity, sort_keys=True, separators=(',', ':')).encode('utf-8')
    identity['run_signature'] = hashlib.sha256(encoded).hexdigest()
    return identity


def prepare_training_run(folder, params, model_kind):
    folder_path = os.path.realpath(folder)
    output_path = os.path.realpath(params['output_path'])
    if folder_path != output_path:
        raise ValueError(
            'job folder and output_path must identify the same directory: '
            '%s != %s' % (folder_path, output_path))

    manifest = build_run_manifest(params, model_kind)
    manifest_path = os.path.join(output_path, RUN_MANIFEST)
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as handle:
            previous = json.load(handle)
        if previous.get('run_signature') != manifest['run_signature']:
            raise RuntimeError(
                'run identity changed; use a new job directory instead of '
                'resuming stale weights')
    elif (os.path.exists(os.path.join(output_path, CHECKPOINT)) or
          os.path.exists(os.path.join(output_path, NETPARAMS))):
        raise RuntimeError(
            'existing weights have no run manifest; move them to a legacy '
            'directory or start this verified run in a new job directory')

    tmp_path = manifest_path + '.tmp'
    with open(tmp_path, 'w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write('\n')
    os.replace(tmp_path, manifest_path)
    return manifest['run_signature']


def require_checkpoint_signature(checkpoint, run_signature):
    if checkpoint.get('run_signature') != run_signature:
        raise RuntimeError(
            'checkpoint identity does not match this code, data, and '
            'configuration; refusing unsafe resume')


def model_state(checkpoint):
    if isinstance(checkpoint, dict) and 'net_state_dict' in checkpoint:
        return checkpoint['net_state_dict']
    return checkpoint


def torch_load(path, device):
    try:
        return torch.load(path, map_location=torch.device(device), weights_only=False)
    except TypeError:
        return torch.load(path, map_location=torch.device(device))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initialize_phase0_model(params, load_initial_checkpoint=True):
    seed_everything(int(params.get('seed', 0)))

    def init_weights(module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)

    net = SubdNet(params)
    net.apply(init_weights)
    initial_checkpoint = params.get('initial_checkpoint')
    if load_initial_checkpoint and initial_checkpoint:
        net.load_state_dict(model_state(torch_load(initial_checkpoint, 'cpu')))
    return net


def require_finite_gradients(net, epoch, mesh_index):
    for name, parameter in net.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(
                'non-finite gradient in %s at epoch %d, mesh %d' %
                (name, epoch, mesh_index))


def atomic_torch_save(data, path):
    tmp_path = path + '.tmp'
    torch.save(data, tmp_path)
    os.replace(tmp_path, path)


def optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def get_rng_state():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state['mps'] = torch.mps.get_rng_state().cpu()
    return state


def set_rng_state(state):
    if not state:
        return
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])
    if 'mps' in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state['mps'].cpu())


def save_checkpoint(path, net, optimizer, next_epoch, best_loss, train_loss_his,
                    valid_loss_his, completed=False, stopped_early=False,
                    run_signature=None, extra_state=None):
    checkpoint = {
        'next_epoch': next_epoch,
        'best_loss': best_loss,
        'train_loss_his': train_loss_his,
        'valid_loss_his': valid_loss_his,
        'net_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'rng_state': get_rng_state(),
        'completed': completed,
        'stopped_early': stopped_early,
        'run_signature': run_signature,
    }
    if extra_state:
        checkpoint.update(extra_state)
    atomic_torch_save(checkpoint, path)


def write_loss_history(params, train_loss_his, valid_loss_his):
    np.savetxt(params['output_path'] + 'train_loss.txt',
               np.array(train_loss_his), delimiter=',')
    np.savetxt(params['output_path'] + 'valid_loss.txt',
               np.array(valid_loss_his), delimiter=',')


def write_validation_outputs(params, T, net):
    net.eval()
    mIdx = 0
    with torch.no_grad():
        x = T.getInputData(mIdx)
        outputs = net(x, mIdx, T.hfList, T.poolMats, T.dofs)

    tgp.writeOBJ(params['output_path'] + str(mIdx) + '_oracle.obj',
                 T.meshes[mIdx][len(outputs) - 1].V.to('cpu'),
                 T.meshes[mIdx][len(outputs) - 1].F.to('cpu'))
    for ii in range(len(outputs)):
        x = outputs[ii].cpu()
        tgp.writeOBJ(params['output_path'] + str(mIdx) + '_subd' + str(ii) + '.obj',
                     x, T.meshes[mIdx][ii].F.to('cpu'))

    with torch.no_grad():
        x = T.getInputData(mIdx)
        dV = torch.rand(1, 3).to(params['device'])
        R = random3DRotation().to(params['device'])
        x[:, :3] = x[:, :3].mm(R.t())
        x[:, 3:] = x[:, 3:].mm(R.t())
        x[:, :3] += dV
        outputs = net(x, mIdx, T.hfList, T.poolMats, T.dofs)

    for ii in range(len(outputs)):
        x = outputs[ii].cpu()
        tgp.writeOBJ(params['output_path'] + str(mIdx) + '_rot_subd' + str(ii) + '.obj',
                     x, T.meshes[mIdx][ii].F.to('cpu'))


def load_best_model_if_available(params, net):
    best_path = params['output_path'] + NETPARAMS
    if os.path.exists(best_path):
        net.load_state_dict(torch_load(best_path, params['device']))
        print('loaded best model for final outputs: %s' % best_path, flush=True)
        return True
    print('warning: best model not found; using current network for final outputs',
          flush=True)
    return False


def main():
    folder = sys.argv[1]
    if not folder.endswith('/'):
        folder += '/'

    with open(folder + 'hyperparameters.json', 'r') as f:
        params = json.load(f)
    os.makedirs(params['output_path'], exist_ok=True)
    run_signature = prepare_training_run(folder, params, 'phase0')

    if params['device'] == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('hyperparameters.json requests cuda, but CUDA is not available')
    if params['device'] == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('hyperparameters.json requests mps, but MPS is not available')

    S = pickle.load(open(params['train_pkl'], 'rb'))
    require_valid_dataset(S, 'training', params['numSubd'])
    S.computeParameters()
    S.toDevice(params['device'])

    T = pickle.load(open(params['valid_pkl'], 'rb'))
    require_valid_dataset(T, 'validation', params['numSubd'])
    T.computeParameters()
    T.toDevice(params['device'])

    checkpoint_path = params['output_path'] + CHECKPOINT
    net = initialize_phase0_model(
        params, load_initial_checkpoint=not os.path.exists(checkpoint_path))
    net = net.to(params['device'])

    lossFunc = torch.nn.MSELoss().to(params['device'])
    optimizer = torch.optim.Adam(net.parameters(), lr=params['lr'])

    trainLossHis = []
    validLossHis = []
    bestLoss = np.inf
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        checkpoint = torch_load(checkpoint_path, params['device'])
        require_checkpoint_signature(checkpoint, run_signature)
        net.load_state_dict(checkpoint['net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        optimizer_to(optimizer, params['device'])
        set_rng_state(checkpoint.get('rng_state'))
        start_epoch = int(checkpoint['next_epoch'])
        bestLoss = checkpoint['best_loss']
        trainLossHis = list(checkpoint.get('train_loss_his', []))
        validLossHis = list(checkpoint.get('valid_loss_his', []))
        print('resuming from checkpoint: epoch %d / %d, best valid %.6e'
              % (start_epoch, params['epochs'], bestLoss), flush=True)
    else:
        print('starting fresh training run', flush=True)
        if params.get('initial_checkpoint'):
            print('loaded matching initial state: %s' %
                  params['initial_checkpoint'], flush=True)

    try:
        for epoch in range(start_epoch, params['epochs']):
            ts = time.time()

            net.train()
            trainErr = 0.0
            for mIdx in range(S.nM):
                x = S.getInputData(mIdx)
                outputs = net(x, mIdx, S.hfList, S.poolMats, S.dofs)

                Vt = S.meshes[mIdx][params['numSubd']].V.to(params['device'])

                loss = 0.0
                for ii in range(params['numSubd'] + 1):
                    nV = outputs[ii].size(0)
                    loss += lossFunc(outputs[ii], Vt[:nV, :])

                if not torch.isfinite(loss).item():
                    raise FloatingPointError(
                        'non-finite train loss at epoch %d, mesh %d' %
                        (epoch, mIdx))

                optimizer.zero_grad()
                loss.backward()
                require_finite_gradients(net, epoch, mIdx)
                optimizer.step()

                loss_value = float(loss.detach().cpu().item())
                if not math.isfinite(loss_value):
                    raise FloatingPointError(
                        'non-finite train loss value at epoch %d, mesh %d' %
                        (epoch, mIdx))
                trainErr += loss_value
            trainLossHis.append(trainErr / S.nM)

            net.eval()
            validErr = 0.0
            with torch.no_grad():
                for mIdx in range(T.nM):
                    x = T.getInputData(mIdx)
                    outputs = net(x, mIdx, T.hfList, T.poolMats, T.dofs)

                    Vt = T.meshes[mIdx][params['numSubd']].V.to(params['device'])

                    loss = 0.0
                    for ii in range(params['numSubd'] + 1):
                        nV = outputs[ii].size(0)
                        loss += lossFunc(outputs[ii], Vt[:nV, :])

                    loss_value = float(loss.detach().cpu().item())
                    if not math.isfinite(loss_value):
                        raise FloatingPointError(
                            'non-finite valid loss at epoch %d, mesh %d' %
                            (epoch, mIdx))
                    validErr += loss_value
            validMean = validErr / T.nM
            validLossHis.append(validMean)

            if validMean < bestLoss:
                bestLoss = validMean
                atomic_torch_save(
                    net.state_dict(), params['output_path'] + NETPARAMS)

            write_loss_history(params, trainLossHis, validLossHis)
            save_checkpoint(checkpoint_path, net, optimizer, epoch + 1, bestLoss,
                            trainLossHis, validLossHis,
                            run_signature=run_signature)

            print('epoch %d, train loss %.6e, valid loss %.6e, remain time: %s'
                  % (epoch, trainLossHis[-1], validLossHis[-1],
                     int(round((params['epochs'] - epoch) * (time.time() - ts)))),
                  flush=True)
            print('checkpoint saved: %s' % checkpoint_path, flush=True)

    except KeyboardInterrupt:
        print('interrupted; latest completed epoch checkpoint is already saved at %s'
              % checkpoint_path, flush=True)
        raise
    except FloatingPointError as exc:
        print('training aborted: %s' % exc, flush=True)
        print('the previous completed-epoch checkpoint remains at %s' %
              checkpoint_path, flush=True)
        raise

    write_loss_history(params, trainLossHis, validLossHis)
    save_checkpoint(checkpoint_path, net, optimizer, params['epochs'], bestLoss,
                    trainLossHis, validLossHis, completed=True,
                    run_signature=run_signature)
    load_best_model_if_available(params, net)
    write_validation_outputs(params, T, net)
    print('training complete; final checkpoint saved: %s' % checkpoint_path, flush=True)


if __name__ == '__main__':
    main()
