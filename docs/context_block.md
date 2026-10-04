# Public Research Scope

This project investigates learning-based triangular-mesh refinement using the released Neural Subdivision implementation as its baseline. Baseline reproduction is complete, and a research extension has been implemented and experimentally evaluated. Further research is ongoing.

Public documentation describes the research problem, implementation infrastructure, validation, and reproducibility workflow. Unpublished architectural mechanisms, detailed variations, and future research designs are intentionally omitted.

## Implemented infrastructure

- Randomized mesh preprocessing and coarse/fine data generation.
- Dataset, topology, correspondence, and finite-value validation.
- Python/PyTorch training with checkpoint/resume and run-identity checks.
- Held-out evaluation and experiment/rendering scripts.
- Structural, baseline parity, rigid-motion/invariance, gradient, and metric tests.

These tools support implementation and experimentation; their presence alone does not establish improved performance or generalization.

## Reproduction and attribution

See the [main README](../README.md) for setup and executable workflow examples, [reproduction notes](reproduction_notes.md) for local data/checkpoint requirements, and [changes from upstream](changes_from_upstream.md) for the contribution boundary.

The original Neural Subdivision method and core implementation belong to their upstream authors. Original licenses are retained; see [third-party notices](../THIRD_PARTY_NOTICES.md).
