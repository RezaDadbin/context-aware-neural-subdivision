# Changes From Upstream

The repository starts from the released Neural Subdivision implementation and
its linked Surface Multigrid remeshing generator. Original licenses and upstream
README files are retained under `external/`.

## Context-conditioned model

`external/neuralSubdiv/context_models.py` adds:

- rigid-motion-invariant per-vertex geometry features;
- topology-restricted local attention;
- chunked whole-mesh attention;
- canonical half-flap context projection; and
- context-conditioned Initialization, Vertex, and Edge predictors.

The original local-frame output, midpoint connectivity, aggregation, and output
ordering remain unchanged.

## Training and evaluation

- `train_context.py` jointly optimizes the Context Block and subdivision core.
- `train_resume.py` adds deterministic initialization, signed run identity,
  resumable optimizer/model state, finite-gradient checks, and best-validation
  checkpoint retention.
- `test_context.py` supports active, disabled, and shuffled context inference.
- `evaluate_model.py` reports correspondence, normal, smooth/detail-region,
  surface-distance, topology, and rigid-motion metrics.
- `validate_dataset.py`, `validate_generated_chain.py`, and
  `smoke_test_dataset.py` gate data before long training runs.

## Core compatibility changes

`models.py` preserves the original I/V/E operations. Its one-ring pooling also
accepts an indexed accumulation representation for Apple MPS, where PyTorch
sparse COO pooling is unavailable. Tests compare the indexed and original CPU
sparse operations.

## Generator changes

The random C++ generator normalizes each input mesh before decimation, matching
the unit-box convention used by the original MATLAB path. The build wrapper and
vendored libigl CMake helper include compatibility settings for current CMake
versions. See `normalization_fix.md`.

## Excluded generated material

Training meshes, PKL datasets, model checkpoints, logs, rendered results,
reports, build directories, and downloaded dependency caches are intentionally
excluded from the public source tree.
