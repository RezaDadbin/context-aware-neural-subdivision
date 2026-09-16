#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GEN_DIR="${ROOT}/external/surface_multigrid_code/09_random_subdiv_remesh"
BUILD_DIR="${GEN_DIR}/build"

echo "============================================================"
echo "Building normalized C++ random subdivision/remeshing generator"
echo "Source: ${GEN_DIR}"
echo "Build:  ${BUILD_DIR}"
echo "============================================================"

cmake -S "${GEN_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build "${BUILD_DIR}" -j "$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"

echo
echo "Generator binary:"
ls -lh "${BUILD_DIR}/random_subdiv_remesh_bin"
