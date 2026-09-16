# C++ Generator Normalization Fix

The original MATLAB data-generation path in Neural Subdivision normalizes the input mesh before
decimation:

```matlab
V = V - min(V);
V = V / max(V(:));
```

This is implemented in `external/neuralSubdiv/utils_matlab/normalizeUnitBox.m`.

For the C++ generator path, I added the same normalization directly after reading the mesh in:

```text
external/surface_multigrid_code/09_random_subdiv_remesh/main.cpp
```

The function is:

```cpp
void normalize_unit_box(Eigen::MatrixXd & V)
```

and it is called before `SSP_random_qslim(...)`. This means generated training meshes are already
in the same unit-box convention expected by the official Python `test.py`, so the separate
`scaled_test.py` workaround is intentionally not included in this cleaned repository.
