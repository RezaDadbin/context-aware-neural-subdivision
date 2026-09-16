# Surface Multigrid Generator Subset

This directory contains the subset of
[`HTDerekLiu/surface_multigrid_code`](https://github.com/HTDerekLiu/surface_multigrid_code)
required by the Neural Subdivision data pipeline:

- `09_random_subdiv_remesh/`: randomized subdivision/remeshing executable;
- `src/`: shared Surface Multigrid source; and
- `libigl/`: the compatible vendored libigl headers and CMake support.

The unrelated upstream demonstration applications are intentionally omitted.
The reference upstream revision is
`adc49abe10a97200a3a5735bd370e98560091a37`. Local normalization and CMake
compatibility changes are documented in `../../docs/changes_from_upstream.md`.

Build from the repository root with:

```bash
./scripts/build_cxx_generator.sh
```

The upstream license is retained in this directory.
