# Context Block Implementation

This codebase implements only the deterministic Context Block of the project
proposal. It does not implement a generative residual, teacher, or prior.

## Data Flow

For each current mesh, the model computes invariant scalar descriptors per
vertex, projects them to 64 dimensions, applies topology-constrained local
attention, and then applies same-mesh global attention. The default experiment
uses six local layers and two global layers with four heads.

The four vertex embeddings of each canonical half-flap are gathered in the
existing `[v0, v1, v2, v3]` order. A projection MLP produces a 64-dimensional
half-flap context. The predictors receive:

```text
[original normalized half-flap descriptor, half-flap context]
```

The original `local2Global`, one-ring/edge pooling, connectivity, and output
ordering are unchanged.

## Invariance

The Context Block never consumes raw world-space XYZ as a feature. It uses
valence, boundary state, normalized edge-length statistics, differential
magnitude, centroid-radius magnitude, normalized face-area statistics, and
dihedral statistics. Attention biases use normalized Euclidean distance,
topological relation, and radial-distance difference.

Local edge and area quantities use mean edge length as their scale. Whole-mesh
radius and global pair distance use RMS distance from the centroid, preventing
their scale from changing simply because a subdivision level has shorter
edges. Dihedral angles are clamped inside the finite-gradient range of `acos`.

Channels after XYZ produced by the original core are scalar channels because
`local2Global` rotates only the first three output channels. These learned
scalar channels are included from the first subdivision transition onward.

## Multi-Level Use

Context is computed before initialization. It is then recomputed from the
current predicted mesh once before every subdivision transition. A transition
reuses one context tensor for its sequential V and E operations. Context is not
carried from one subdivision level to the next.

## Phase-0 Compatibility

`ContextConditionedMLP` copies every original I/V/E weight and bias. New first
layer columns multiplying context are initialized to zero. Passing
`context_enabled=False` bypasses those columns, allowing direct numerical
comparison against `SubdNet`.

Fresh context training also saves `phase0_initial_state.dat` in the job folder.
That is the matching untrained Phase-0 initialization for a controlled later
comparison; it is not a previously trained model.

The Phase-0 `train_resume.py` accepts this path as `initial_checkpoint`, so the
shared I/V/E parameters can be loaded rather than merely saved.

## Training Reliability

The core keeps the original learning rate. The Context Block and half-flap
context projection use the separately configurable `context_lr`, which defaults
to `2e-4`. Both groups are optimized jointly from the same cross-level loss.
All gradients are checked for finite values before every optimizer step.
Because zero context-input columns intentionally block Context Block gradients
on the first step, training explicitly verifies nonzero Context Block gradients
on the second step. A failure aborts the run.

On Apple MPS, one-ring pooling uses indexed accumulation because sparse COO
matrix multiplication is unavailable. A parity test verifies the same result
as the original CPU sparse operation. Checkpoints also save and restore MPS RNG
state.

Every run signs its data files, starting weights, architecture, learning
settings, seed, and model source files. Resume is refused if that identity
changes. The resumable checkpoint retains a matching final model/optimizer
pair, while `netparams.dat` stores the best-validation model for evaluation.

`validate_dataset.py` checks exact official midpoint connectivity/order,
topology, winding, isolated vertices, finite coordinates, target-prefix
correspondence, core-frame conditioning, and degenerate/narrow faces before
training. `evaluate_model.py` reports correspondence, sampled surface, normal,
topology, mesh-quality, smooth/detail-region, and rigid-motion metrics after
training. Reliable self-intersection testing remains an external requirement.

## Configuration

No dataset is selected in this codebase. Create a job only after choosing PKLs:

```bash
python scripts/write_context_hyperparams.py EXPERIMENT_NAME \
  --train-pkl /path/to/train.pkl \
  --valid-pkl /path/to/valid.pkl
./scripts/train_context.sh EXPERIMENT_NAME
```

The local/global layer counts and global query chunk size are configurable.
Setting `context_global_layers` to zero provides the planned local-only
ablation. `test_context.py --shuffle-context` and `--disable-context` provide
the shuffled-context and compatibility paths.
