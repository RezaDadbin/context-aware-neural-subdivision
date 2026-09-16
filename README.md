# Context-Aware Neural Subdivision

Research implementation of a context-conditioned extension to Neural
Subdivision. The released Initialization, Vertex, and Edge (I/V/E) subdivision
core is preserved, while a deterministic Context Block supplies wider
topology-local and whole-mesh information to each predictor.

This repository contains source code and reproducibility tooling. Datasets,
source meshes, trained checkpoints, logs, renders, and report files are not
included.

## Architecture

The Context Block:

1. computes rigid-motion-invariant scalar features on the current mesh;
2. applies six topology-restricted local-attention layers;
3. applies two chunked whole-mesh attention layers;
4. projects the four contextual half-flap embeddings into a 64-dimensional
   conditioning vector; and
5. concatenates that vector with the original canonical half-flap descriptor
   before the I/V/E predictors.

Context and subdivision parameters are optimized jointly. Connectivity,
canonical local frames, local-to-global conversion, aggregation, and vertex
ordering remain part of the original subdivision core.

## Repository Layout

```text
configs/                         dataset-neutral model configuration
docs/                            architecture and reproduction notes
external/neuralSubdiv/           preserved core and context implementation
external/surface_multigrid_code/ C++ subdivision/remeshing generator source
scripts/                         generation, training, evaluation, and rendering
tests/                           parity, invariance, topology, and metric tests
```

The main implementation files are:

```text
external/neuralSubdiv/context_models.py
external/neuralSubdiv/train_context.py
external/neuralSubdiv/test_context.py
external/neuralSubdiv/evaluate_model.py
external/neuralSubdiv/validate_dataset.py
external/neuralSubdiv/validate_generated_chain.py
external/neuralSubdiv/train_resume.py
```

## Installation

Python 3.11 was used for the reported experiments.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Run the source-only test suite:

```bash
python3 -m unittest discover -s tests -v
```

## Data Preparation

The experiment scripts expect legally obtained, closed triangular source meshes
at the following paths:

```text
data_candidates/compact_context/01_bimba_head.obj
data_candidates/compact_context/02_fat_dragon.obj
data_candidates/compact_context/03_gear16.obj
```

The mesh files are intentionally excluded because redistribution rights depend
on their original datasets. Substitute other compatible meshes by updating the
paths in `scripts/generate_all_context_datasets.sh`.

Build the C++ generator:

```bash
./scripts/build_cxx_generator.sh
```

The first CMake configuration may download third-party build dependencies.
Then generate and validate all configured splits:

```bash
./scripts/generate_all_context_datasets.sh
```

Each accepted chain follows `500 -> 2,000 -> 8,000` faces. The generator keeps
sampling seeds until the requested train, validation, and held-out splits pass
the topology, correspondence, finite-value, and core-frame checks.

## Training

Create one context job explicitly:

```bash
python3 scripts/write_context_hyperparams.py EXPERIMENT_NAME \
  --train-pkl /path/to/train.pkl \
  --valid-pkl /path/to/valid.pkl
./scripts/train_context.sh EXPERIMENT_NAME
```

The project-specific six-model runner is also retained:

```bash
PYTHON_BIN=python3 ./scripts/train_all_six_models_700.sh
```

Training saves a resumable checkpoint after each completed epoch and retains
the lowest-validation-loss weights in `netparams.dat`.

## Evaluation

Evaluate a context-conditioned checkpoint:

```bash
cd external/neuralSubdiv
python3 evaluate_model.py /path/to/job /path/to/held_out.pkl \
  --model context --modes context disabled shuffled
```

The intervention modes test whether the trained model depends on correctly
aligned context. See `docs/context_block.md` for implementation details and
`docs/reproduction_notes.md` for artifact and dataset expectations.

## Scope

The current experiments use one object-specific model per source mesh. They
evaluate object-specific structural context and do not establish cross-object
semantic generalization. The proposed stochastic Generative Block is not
implemented in this repository.

## Attribution and License

This work modifies and extends the released Neural Subdivision implementation
and uses the linked Surface Multigrid subdivision/remeshing code. Upstream
licenses are preserved in their respective directories. See
`THIRD_PARTY_NOTICES.md` and `docs/changes_from_upstream.md`.

Source files covered by Neural Subdivision's license remain under the Mozilla
Public License 2.0; see `LICENSE`.
