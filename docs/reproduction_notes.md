# Reproduction Notes

## Included

- the Neural Subdivision core and Context Block implementation;
- source for the official linked random subdivision/remeshing generator;
- dataset validation and smoke tests;
- resumable Phase 0 and Phase 1 training launchers;
- held-out evaluation and context-intervention tools; and
- unit tests for parity, invariance, topology, gradients, and metrics.

## Not included

- source meshes or third-party datasets;
- generated OBJ chains and PKL files;
- trained checkpoints and optimizer state;
- logs, renders, report figures, or PDFs; and
- CMake build products and downloaded dependency caches.

These files are excluded because they are generated, machine-specific, large,
or subject to separate dataset redistribution terms.

## Expected experiment data

The project-specific runner expects 200 training, 20 validation, and 20
held-out chains per source object. Every chain contains exactly 500, 2,000, and
8,000 faces with the midpoint-subdivision topology and vertex correspondence
required by the Neural Subdivision loss.

Place compatible source meshes under `data_candidates/compact_context/` or edit
the source map in `scripts/generate_all_context_datasets.sh`. The script builds
each split, validates every generated level, creates PKL files, and runs Phase 0
and Phase 1 forward/backward smoke tests before training.

## Checkpoint behavior

`checkpoint_latest.pt` stores the resumable model, optimizer, epoch, run
identity, and random-number-generator state. `netparams.dat` stores the model
with the lowest validation loss and is the checkpoint used for final
evaluation. Rerunning the project training launcher skips completed models and
resumes incomplete ones.
