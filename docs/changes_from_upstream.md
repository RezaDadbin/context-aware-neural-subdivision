# Changes From Upstream

This repository starts from the released [Neural Subdivision implementation](https://github.com/HTDerekLiu/neuralSubdiv) and its linked [Surface Multigrid remeshing generator](https://github.com/HTDerekLiu/surface_multigrid_code). Original licenses and upstream README files are retained under `external/`. Reference revisions are listed in [third-party notices](../THIRD_PARTY_NOTICES.md).

## Research implementation

The project reproduces the baseline workflow and includes an implemented, experimentally evaluated extension. Public documentation records this contribution and its status without describing unpublished mechanisms. The original Neural Subdivision method is not an original contribution of this repository.

## Training, validation, and evaluation infrastructure

- Experiment configuration and data-generation launchers.
- Resumable model/optimizer state, random-number-generator state, run-identity checks, finite-gradient checks, and best-validation checkpoint retention.
- Dataset and generated-chain validation before long training runs.
- Held-out evaluation for correspondence, surface distance, normals, topology, mesh quality, and rigid-motion behavior.
- Structural, parity, invariance, topology, gradient, and metric tests.

## Compatibility and generator changes

The baseline's one-ring pooling supports indexed accumulation for Apple MPS, where sparse COO pooling is unavailable. Tests compare the indexed and original CPU sparse operations.

The random C++ generator normalizes each input mesh before decimation, matching the unit-box convention in the original MATLAB path. Build wrappers and a vendored libigl CMake helper contain compatibility settings for newer CMake versions. See [the normalization note](normalization_fix.md).

## Excluded generated material

Training meshes, PKL datasets, model checkpoints, logs, renders, reports, build directories, and downloaded dependency caches are excluded from the tracked public source tree. Their exclusion reflects generated-file size, machine-specific state, or separate dataset redistribution terms.
