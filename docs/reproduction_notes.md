# Reproduction Notes

## Included infrastructure

- An upstream-derived Neural Subdivision research implementation, with a reproduced baseline and an experimentally evaluated extension.
- Source for the linked random subdivision/remeshing generator.
- Dataset validation and forward/backward smoke tests.
- Resumable training, held-out evaluation, and experiment scripts.
- Unit tests for parity, invariance, topology, gradients, and metrics.

The public documentation summarizes infrastructure and workflow. Unpublished mechanisms and unreleased research designs are outside its scope.

## Local data and artifacts

The repository excludes source meshes, third-party datasets, generated OBJ chains and PKL files, trained checkpoints and optimizer state, logs, renders, report figures/PDFs, CMake build products, and downloaded dependency caches. These files are generated, machine-specific, large, or subject to separate redistribution terms.

## Supplied data-generation workflow

The supplied object-specific runner uses 200 training, 20 validation, and 20 held-out chains per source mesh. Each chain contains 500, 2,000, and 8,000 faces with the midpoint-subdivision topology and vertex correspondence required by the upstream Neural Subdivision loss. These are data-generation settings, not performance results.

Place legally obtained, compatible meshes under `data_candidates/compact_context/` or update the source map in `scripts/generate_all_context_datasets.sh`. The script builds the splits, validates generated levels, creates PKLs, and performs forward/backward smoke tests before training.

## Checkpoint behavior

`checkpoint_latest.pt` stores resumable model, optimizer, epoch, run identity, and random-number-generator state. `netparams.dat` stores the lowest-validation-loss model used for evaluation. Use the same data and compatible settings when resuming a job.

Record source-mesh provenance, split definitions, seeds, package versions, hardware, experiment configuration, and the repository commit. A public source release does not reproduce private data or excluded experiment artifacts by itself.
