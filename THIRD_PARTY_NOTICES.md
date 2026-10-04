# Third-Party Notices

This repository contains modified or redistributed source from the following
projects. Their original license files are retained with the source.

## Neural Subdivision

- Project: https://github.com/HTDerekLiu/neuralSubdiv
- Local path: `external/neuralSubdiv`
- Reference revision: `fa64a7384cd375b4c26cedf3f1b86ac2f5ab2e16`
- License: Mozilla Public License 2.0

The research extension, resumable training, validation, evaluation, and MPS
compatibility contributions are summarized in [changes from upstream](docs/changes_from_upstream.md).
The original Neural Subdivision method and core implementation are attributed
to their upstream authors; see the [paper/project page](https://www.dgp.toronto.edu/projects/neural-subdivision/).

## Surface Multigrid via Intrinsic Prolongation

- Project: https://github.com/HTDerekLiu/surface_multigrid_code
- Local path: `external/surface_multigrid_code`
- Reference revision: `adc49abe10a97200a3a5735bd370e98560091a37`
- License: see `external/surface_multigrid_code/LICENSE`

The included random subdivision/remeshing generator has the normalization and
build-compatibility changes described in `docs/normalization_fix.md` and
`docs/changes_from_upstream.md`.

## libigl

- Project: https://github.com/libigl/libigl
- Local path: `external/surface_multigrid_code/libigl`
- Licenses: see `LICENSE.MPL2` and `LICENSE.GPL` in that directory

Downloaded build dependencies are not committed to this repository and retain
their respective upstream licenses.
