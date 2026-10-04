# Context-Aware Neural Subdivision

## Overview

Research implementation built around [Neural Subdivision](https://www.dgp.toronto.edu/projects/neural-subdivision/) for learning-based triangular-mesh refinement. The project includes a reproduced baseline pipeline and an implemented, experimentally evaluated research extension. Further research is ongoing.

The public documentation focuses on implementation and reproducibility infrastructure:

- Randomized mesh preprocessing and coarse/fine hierarchy generation.
- Topology, correspondence, and dataset validation before training.
- Python/PyTorch training, checkpoint/resume, and experiment management.
- Held-out evaluation, mesh-quality checks, and rendering tools.
- Baseline parity, rigid-motion/invariance, topology, and metric tests.

Datasets, source meshes, trained checkpoints, logs, renders, and report files are not included. Details of ongoing unreleased research are intentionally omitted from the public documentation.

## Repository Structure

```text
configs/                         dataset-neutral model configuration
docs/                            scope, attribution, and reproduction notes
external/neuralSubdiv/           upstream-derived research implementation
external/surface_multigrid_code/ C++ subdivision/remeshing generator source
scripts/                         data generation, training, evaluation, rendering
tests/                           structural, invariance, parity, and metric tests
```

For the contribution boundary, see [Changes from upstream](docs/changes_from_upstream.md). For data and checkpoint expectations, see [Reproduction notes](docs/reproduction_notes.md).

## Installation

Use Python 3.11 and a virtual environment. Building the data generator also requires CMake and a C++ compiler.

```bash
git clone https://github.com/RezaDadbin/context-aware-neural-subdivision.git
cd context-aware-neural-subdivision
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`; the shell launchers require a Bash-compatible environment. Select a PyTorch build appropriate for your hardware when preparing the environment.

## Data Preparation

The supplied generation workflow expects legally obtained, closed triangular source meshes at:

```text
data_candidates/compact_context/01_bimba_head.obj
data_candidates/compact_context/02_fat_dragon.obj
data_candidates/compact_context/03_gear16.obj
```

Source meshes are excluded because redistribution rights depend on their original datasets. To use other compatible meshes, update the source paths in `scripts/generate_all_context_datasets.sh`.

From the repository root:

```bash
./scripts/build_cxx_generator.sh
./scripts/generate_all_context_datasets.sh
```

The first CMake configuration may download third-party build dependencies. Generation produces randomized coarse/fine chains and checks topology, correspondence, finite values, and local-frame validity before creating training, validation, and held-out PKL splits. See [Reproduction notes](docs/reproduction_notes.md) for the supplied data layout.

## Training

Create an experiment using your generated datasets. Replace the paths below with absolute paths to your PKL files:

```bash
python3 scripts/write_context_hyperparams.py EXPERIMENT_NAME \
  --train-pkl /path/to/train.pkl \
  --valid-pkl /path/to/valid.pkl
./scripts/train_context.sh EXPERIMENT_NAME
```

The job is written under `external/neuralSubdiv/jobs/net_context_EXPERIMENT_NAME/`. Training saves a resumable checkpoint after each completed epoch and retains the lowest-validation-loss model in `netparams.dat`. Rerun the same launcher to resume an interrupted job with compatible data and settings.

## Evaluation

After training, evaluate the saved model on a separate held-out split:

```bash
cd external/neuralSubdiv
python3 evaluate_model.py /path/to/job /path/to/held_out.pkl \
  --model context --modes context
```

The evaluator provides correspondence, surface-distance, normal, topology, mesh-quality, and rigid-motion checks. Evaluation requires local datasets and a trained checkpoint; neither is bundled. No unpublished quantitative results are reported here.

## Testing

From the repository root, with the Python dependencies installed:

```bash
python3 -m unittest discover -s tests -v
```

The tests cover structural behavior, baseline compatibility, rigid-motion/invariance, topology, finite gradients, and metric calculations. They do not require the excluded research datasets or trained checkpoints.

## Reproducibility

Record the repository commit, Python/package versions, hardware/device, source-mesh provenance, dataset splits, random seed, and experiment configuration. Keep generated data, checkpoints, logs, and evaluation outputs together outside the tracked source tree.

The training infrastructure records run identity and random-number-generator state for resume checks. Reproducibility still depends on compatible data, software, and hardware; identical seeds do not guarantee identical results across platforms.

## Attribution / Third-Party Code

The original **Neural Subdivision** method and released implementation are by Hsueh-Ti Derek Liu, Vladimir G. Kim, Siddhartha Chaudhuri, Noam Aigerman, and Alec Jacobson (SIGGRAPH 2020). This repository reproduces and extends their work; it does not claim authorship of the original method.

- [Neural Subdivision paper/project](https://www.dgp.toronto.edu/projects/neural-subdivision/) and [upstream code](https://github.com/HTDerekLiu/neuralSubdiv).
- [Surface Multigrid subdivision/remeshing code](https://github.com/HTDerekLiu/surface_multigrid_code), used for data generation.
- [Third-party notices](THIRD_PARTY_NOTICES.md), including upstream reference revisions and license locations.

Original license files and upstream README files are preserved under `external/`.

## Research Status / Scope

Baseline reproduction is complete; a research extension has been implemented and experimentally evaluated. Further research is ongoing. Current workflows train separate models per source mesh and do not establish cross-object semantic generalization. This documentation describes the research problem, implemented infrastructure, and status without specifying unpublished mechanisms or future architectures.
